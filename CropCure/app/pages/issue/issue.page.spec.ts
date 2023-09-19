import { ComponentFixture, TestBed } from '@angular/core/testing';
import { IssuePage } from './issue.page';

describe('IssuePage', () => {
  let component: IssuePage;
  let fixture: ComponentFixture<IssuePage>;

  beforeEach(async(() => {
    fixture = TestBed.createComponent(IssuePage);
    component = fixture.componentInstance;
    fixture.detectChanges();
  }));

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
