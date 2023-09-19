import { ComponentFixture, TestBed } from '@angular/core/testing';
import { ExpertPagePage } from './expert-page.page';

describe('ExpertPagePage', () => {
  let component: ExpertPagePage;
  let fixture: ComponentFixture<ExpertPagePage>;

  beforeEach(async(() => {
    fixture = TestBed.createComponent(ExpertPagePage);
    component = fixture.componentInstance;
    fixture.detectChanges();
  }));

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
