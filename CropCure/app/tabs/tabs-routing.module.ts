import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { TabsPage } from './tabs.page';

const routes: Routes = [
  {
    path: 'tabs',
    component: TabsPage,
    children: [
      {
        path: 'issue',
        loadChildren: () => import('../pages/issue/issue.module').then(m => m.IssuePageModule)
      },
      {
        path: 'report',
        loadChildren: () => import('../pages/report/report.module').then(m => m.ReportPageModule)
      },
      {
        path: 'expert-page',
        loadChildren: () => import('../pages/expert-page/expert-page.module').then(m => m.ExpertPagePageModule)
      },
      {
        path: 'chat',
        loadChildren: () => import('../pages/chat/chat.module').then(m => m.ChatPageModule)
      },
      {
        path: 'report',
        loadChildren: () => import('../pages/report/report.module').then(m => m.ReportPageModule)
      },
      {
        path: 'farmer',
        loadChildren: () => import('../pages/farmer/farmer.module').then(m => m.FarmerPageModule)
      },
      {
        path: 'view',
        loadChildren: () => import('../pages/view/view.module').then(m => m.ViewPageModule)
      },
    ]
  },
  {
    path: '',
    redirectTo: '/tabs/tab1',
    pathMatch: 'full'
  }
];

@NgModule({
  imports: [RouterModule.forChild(routes)],
})
export class TabsPageRoutingModule {}
